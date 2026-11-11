"""Read simulation parameters and preserve native interface source text."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError


PARAMETER_DOMAINS = (
    "base", "buses", "branches", "sources", "units", "elements", "signals",
    "events", "simulation", "meta",
)

_VIEW_HEADER = re.compile(r"^@([A-Za-z_][\w-]*)(?:\s+#.*)?\s*$")
_VIEW_DEFINITION = re.compile(
    r"^[A-Za-z_][\w-]*\s*:\s*[A-Za-z_][\w.-]*(?:\s*,\s*[A-Za-z_][\w.-]*)?\s*$"
)
_REFERENCE = r"[A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*)?"
_GROUP = re.compile(rf"^&[A-Za-z_][\w-]*\s*:\s*&?{_REFERENCE}(?:\s*,\s*&?{_REFERENCE})*\s*$")
_CONNECTION = re.compile(rf"^{_REFERENCE}(?:\s*(?:--|->)\s*{_REFERENCE})+\s*$")
_KEY = re.compile(r"^([A-Za-z_][\w-]*|'(?:[^']|'')*'|\"(?:[^\"\\]|\\.)*\")\s*:\s*(.*)$")
_READERS: list[Callable[[str, Path], Mapping[str, Any] | None]] = []


class SimulationFileError(ValueError):
    """Invalid native syntax or an unavailable source-format addon."""

    def __init__(self, message: str, *, path: str | Path | None = None,
                 line: int = 0, column: int = 0) -> None:
        self.message, self.path = message, path
        self.line, self.column = line, column
        location = (f" at line {line}, column {column}" if column else
                    f" at line {line}" if line else "")
        prefix = f"{path}: " if path is not None else ""
        super().__init__(f"{prefix}{message}{location}")


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects a key instead of silently replacing it."""


def _construct_mapping(loader: _UniqueKeyLoader, node, deep: bool = False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError("while constructing a mapping", node.start_mark,
                                   "found an unhashable key", key_node.start_mark) from exc
        if duplicate:
            raise ConstructorError("while constructing a mapping", node.start_mark,
                                   f"duplicate key: {key}", key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping,
)


def register_simulation_reader(
        reader: Callable[[str, Path], Mapping[str, Any] | None],
) -> Callable[[str, Path], Mapping[str, Any] | None]:
    """Register one fallback reader for a non-native simulation-file format."""
    if reader not in _READERS:
        _READERS.append(reader)
    return reader


def is_native_source(text: str) -> bool:
    """Whether a source buffer contains top-level native interface syntax."""
    return any(line.startswith("@") for line in text.removeprefix("\ufeff").splitlines())


def _location(exc: Exception) -> tuple[int, int]:
    mark = getattr(exc, "problem_mark", None)
    if mark is not None:
        return mark.line + 1, mark.column + 1
    if isinstance(exc, json.JSONDecodeError):
        return exc.lineno, exc.colno
    return 0, 0


def parse_parameter_mapping(text: str, path: str | Path = "<string>", *,
                            domains: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Parse a JSON or YAML mapping, rejecting duplicate and unknown keys when requested."""
    try:
        if Path(path).suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.load(text, Loader=_UniqueKeyLoader)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        message = getattr(exc, "problem", None) or getattr(exc, "msg", None)
        message = message or str(exc).splitlines()[0]
        line, column = _location(exc)
        raise SimulationFileError(message, path=path, line=line, column=column) from None
    if not isinstance(data, dict):
        raise SimulationFileError("top level must be a mapping", path=path)
    if domains is not None:
        unknown = next((key for key in data if key not in domains), None)
        if unknown is not None:
            raise SimulationFileError(
                f"unknown parameter domain: {unknown}; known: {', '.join(domains)}", path=path,
            )
    return data


def _blank(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith(("\n", "\r")):
        return line[-1]
    return ""


def _content(line: str) -> str:
    """Remove a trailing comment without interpreting view expressions."""
    quote = ""
    escaped = False
    for index, character in enumerate(line):
        if quote:
            if quote == '"' and character == "\\" and not escaped:
                escaped = True
                continue
            if character == quote and not escaped:
                if quote == "'" and index + 1 < len(line) and line[index + 1] == "'":
                    continue
                quote = ""
            escaped = False
        elif character in "'\"":
            quote = character
        elif character == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index].rstrip()
    return line.rstrip()


def _starts_parameters(statement: str) -> bool:
    """Recognize a parameter domain without treating a bare view declaration as one."""
    content = _content(statement).strip()
    if content in {"---", "..."}:
        return True
    match = _KEY.fullmatch(content)
    if match is None:
        return False
    written, value = match.groups()
    if written[:1] in "'\"":
        return True
    if written in PARAMETER_DOMAINS:
        return True
    return bool(value.strip()) and not _VIEW_DEFINITION.fullmatch(content)


def _validate_view(name: str, body: list[tuple[int, str]], path: str | Path) -> None:
    """Validate the common structural grammar; rendering remains a GUI concern."""
    declaration_indent: int | None = None
    for number, raw in body:
        statement = raw.rstrip("\r\n")
        if not statement.strip() or statement.lstrip().startswith("#"):
            continue
        prefix = statement[:len(statement) - len(statement.lstrip(" \t"))]
        if "\t" in prefix:
            raise SimulationFileError("tab indentation is not supported", path=path, line=number)
        level = len(prefix)
        expression = _content(statement).strip()
        if not expression:
            continue
        if level:
            if declaration_indent is None or level <= declaration_indent:
                raise SimulationFileError(
                    "indented content requires a declaration", path=path, line=number,
                )
            continue
        declaration_indent = None
        if _GROUP.fullmatch(expression) or _CONNECTION.fullmatch(expression):
            continue
        if _VIEW_DEFINITION.fullmatch(expression):
            declaration_indent = level
            continue
        if re.fullmatch(r"[A-Za-z_][\w-]*\s*:\s*", expression):
            raise SimulationFileError(
                f"{expression.split(':', 1)[0]}: requires a displayed type",
                path=path, line=number,
            )
        raise SimulationFileError(f"invalid statement in @{name} view", path=path, line=number)


def split_simulation_text(text: str, path: str | Path = "<string>") -> tuple[str, str]:
    """Split native source into a line-preserving parameter body and raw interface text."""
    source = text.removeprefix("\ufeff")
    lines = source.splitlines(keepends=True)
    parameters = list(lines)
    interface: list[str] = []
    names: set[str] = set()
    active_name: str | None = None
    active_start = 0

    def finish(end: int) -> None:
        nonlocal active_name
        if active_name is None:
            return
        _validate_view(
            active_name,
            [(index + 1, lines[index]) for index in range(active_start + 1, end)],
            path,
        )
        interface.append("".join(lines[active_start:end]).rstrip("\r\n"))
        active_name = None

    for index, line in enumerate(lines):
        statement = line.rstrip("\r\n")
        if statement.startswith("@"):
            finish(index)
            match = _VIEW_HEADER.fullmatch(statement)
            if match is None:
                raise SimulationFileError(
                    "invalid interface view header", path=path, line=index + 1,
                )
            name = match.group(1)
            if name in names:
                raise SimulationFileError(
                    f"duplicate @{name} view", path=path, line=index + 1,
                )
            names.add(name)
            active_name, active_start = name, index
            parameters[index] = _blank(line)
        elif active_name is not None and not statement[:1].isspace() and _starts_parameters(statement):
            finish(index)
        elif active_name is not None:
            parameters[index] = _blank(line)
    finish(len(lines))
    if "circuit" not in names:
        raise SimulationFileError("native simulation files require an @circuit view", path=path)
    return "".join(parameters), "\n\n".join(interface) + "\n"


def parse_native_text(text: str, path: str | Path = "<string>") -> tuple[dict[str, Any], str]:
    """Return the parameter mapping and untouched interface source of a native file."""
    parameters, interface = split_simulation_text(text, path)
    return parse_parameter_mapping(parameters, path, domains=PARAMETER_DOMAINS), interface


def parse_simulation_text(text: str, path: str | Path = "<string>") -> tuple[dict[str, Any], str]:
    """Parse native syntax, or delegate a parameter-only buffer to a format addon."""
    source = Path(path)
    if is_native_source(text):
        return parse_native_text(text, source)
    errors: list[SimulationFileError] = []
    for reader in _READERS:
        try:
            parameters = reader(text, source)
        except SimulationFileError as exc:
            errors.append(exc)
            continue
        if parameters is not None:
            unknown = next((key for key in parameters if key not in PARAMETER_DOMAINS), None)
            if unknown is not None:
                raise SimulationFileError(
                    f"unknown parameter domain: {unknown}; known: {', '.join(PARAMETER_DOMAINS)}",
                    path=path,
                )
            return dict(parameters), ""
    if errors:
        raise errors[0]
    raise SimulationFileError(
        "native simulation files require an @circuit view; "
        "this source needs a compatible format addon", path=path,
    )


def read_simulation_file(path: str | Path) -> tuple[dict[str, Any], str]:
    """Read a simulation file as parameters plus optional raw interface text."""
    source = Path(path)
    return parse_simulation_text(source.read_text(encoding="utf-8"), source)


__all__ = [
    "PARAMETER_DOMAINS", "SimulationFileError", "is_native_source", "parse_native_text",
    "parse_parameter_mapping", "parse_simulation_text", "read_simulation_file",
    "register_simulation_reader", "split_simulation_text",
]
