"""Checked file operations for the frontend host adapters.

All file paths are relative to one explicitly selected directory.  The source
text and its on-disk version are the only editing state: no shadow schematic is
saved, and graphical patches replace exactly one YAML scalar span.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

TEXT_SUFFIXES = frozenset({".pes", ".yaml", ".yml", ".json"})
READ_SUFFIXES = TEXT_SUFFIXES | {".csv"}
MAX_FILE_SIZE = 4 * 1024 * 1024
MAX_ROWS = 4000
MAX_SCANNED_ROWS = 20000
MAX_CSV_RECORD_BYTES = 1024 * 1024
COMPONENT_SECTIONS = frozenset({"buses", "sources", "branches", "elements", "units"})
FILE_SECTIONS = ("base", "buses", "branches", "sources", "units", "elements",
                 "signals", "events", "simulation", "meta")


class WorkspaceError(ValueError):
    """A rejected client operation; HTTP adapters use its status code."""

    status = 400


class FileConflict(WorkspaceError):
    """The file differs from the version the client read."""

    status = 409


def content_version(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scalar_text(value: Any) -> str:
    """Emit a YAML scalar without allowing arbitrary YAML structure or NaN."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    raise WorkspaceError("graphical edits must specify a finite scalar")


def _node_at_path(text: str, path: str) -> ScalarNode:
    """Find an unambiguous scalar in source coordinates, including flow mappings."""
    if not path or any(not part for part in path.split(".")):
        raise WorkspaceError("parameter path must contain nonempty dotted keys")
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise WorkspaceError(f"invalid YAML: {exc}") from exc
    if not isinstance(root, MappingNode):
        raise WorkspaceError("the .pes file must contain a mapping")
    current = root
    # Any aliases or merge keys on the selected path are ambiguous for source editing.
    visited: set[int] = set()
    for key in path.split("."):
        if not isinstance(current, MappingNode) or id(current) in visited:
            raise WorkspaceError("ambiguous or non-mapping parameter path")
        visited.add(id(current))
        matches = [value for k, value in current.value
                   if isinstance(k, ScalarNode) and k.value == key]
        if len(matches) != 1:
            raise WorkspaceError(f"missing or duplicated YAML key: {key}")
        current = matches[0]
    if not isinstance(current, ScalarNode) or current.style in ("|", ">"):
        raise WorkspaceError("only explicit, single-line scalar values can be edited")
    # YAML aliases share the *same* node.  An edit must not modify aliased peers.
    references: dict[int, int] = {}
    def scan(node: Any, ancestors: frozenset[int]) -> None:
        references[id(node)] = references.get(id(node), 0) + 1
        if id(node) in ancestors:
            return
        more = ancestors | {id(node)}
        if isinstance(node, MappingNode):
            for k, v in node.value:
                scan(k, more)
                scan(v, more)
        elif isinstance(node, SequenceNode):
            for child in node.value:
                scan(child, more)
    scan(root, frozenset())
    if references.get(id(current), 0) != 1:
        raise WorkspaceError("aliased values must be edited in source text")
    if current.start_mark.index >= current.end_mark.index or (
            current.start_mark.line != current.end_mark.line):
        raise WorkspaceError("empty or multiline values must be edited in source text")
    return current


def patch_scalar(text: str, path: str, value: Any) -> dict[str, Any]:
    """Return a single source-range edit.  Do not rewrite unrelated YAML."""
    node = _node_at_path(text, path)
    return {
        "start": node.start_mark.index,
        "end": node.end_mark.index,
        "replacement": scalar_text(value),
        "line": node.start_mark.line + 1,
        "column": node.start_mark.column + 1,
    }


def patched_text(text: str, patch: dict[str, Any]) -> str:
    return text[:patch["start"]] + patch["replacement"] + text[patch["end"]:]


def explicit_parameters(text: str) -> list[dict[str, Any]]:
    """List scalar paths and source spans without losing original spelling."""
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise WorkspaceError(f"invalid YAML: {exc}") from exc
    results: list[dict[str, Any]] = []
    def walk(node: Any, parents: tuple[str, ...], visited: frozenset[int]) -> None:
        if not isinstance(node, MappingNode) or id(node) in visited:
            return
        seen: set[str] = set()
        for k, v in node.value:
            if not isinstance(k, ScalarNode) or k.value in seen or "." in k.value:
                continue
            seen.add(k.value)
            next_path = parents + (k.value,)
            if isinstance(v, ScalarNode) and v.start_mark.line == v.end_mark.line:
                results.append({"path": ".".join(next_path), "text": v.value,
                                "start": v.start_mark.index, "end": v.end_mark.index,
                                "line": v.start_mark.line + 1,
                                "column": v.start_mark.column + 1})
            else:
                walk(v, next_path, visited | {id(node)})
    walk(root, (), frozenset())
    return results



def _mapping_document(text: str) -> MappingNode:
    """Parse one editable YAML document without discarding source coordinates."""
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise WorkspaceError(f"invalid YAML: {exc}") from exc
    if not isinstance(root, MappingNode) or root.flow_style:
        raise WorkspaceError("component editing requires a block-style YAML document")
    return root


def _entry(node: MappingNode, name: str) -> Any | None:
    matches = [value for key, value in node.value
               if isinstance(key, ScalarNode) and key.value == name]
    if len(matches) > 1:
        raise WorkspaceError(f"duplicated YAML key: {name}")
    return matches[0] if matches else None


def add_component(text: str, section: str, name: str,
                  definition: str) -> dict[str, Any]:
    """Insert one mapping entry without reserializing any existing file content.

    A new section is inserted in conventional PESLite file order. Existing
    flow-style mappings, aliases, ambiguous keys and custom unsupported
    configurations are never rewritten to accommodate insertion.
    """
    if section not in COMPONENT_SECTIONS:
        raise WorkspaceError("unsupported component section")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise WorkspaceError("component names must be simple PESLite identifiers")
    if not isinstance(definition, str) or len(definition) > 65536:
        raise WorkspaceError("component definition must be YAML text under 64 KiB")
    try:
        values = yaml.safe_load(definition)
    except yaml.YAMLError as exc:
        raise WorkspaceError(f"invalid component YAML: {exc}") from exc
    if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
        raise WorkspaceError("component definition must be a YAML mapping")
    root = _mapping_document(text)
    target = _entry(root, section)
    if target is not None:
        if not isinstance(target, MappingNode) or target.flow_style:
            raise WorkspaceError("cannot add a component to a flow-style or non-mapping section")
        if _entry(target, name) is not None:
            raise WorkspaceError(f"component {name!r} already exists in {section}")
        index = target.end_mark.index
        # PyYAML marks a block's end at the next declaration. A flow-style
        # last child may end before an inline comment: insert after its line.
        if target.end_mark.column:
            newline = text.find("\n", index)
            index = newline + 1 if newline >= 0 else len(text)
        content = yaml.safe_dump({name: values}, sort_keys=False,
                                 allow_unicode=True, default_flow_style=False)
        replacement = ("" if not index or text[index - 1] == "\n" else "\n")
        replacement += "".join("  " + line + "\n" for line in content.splitlines())
    else:
        insertion_point = len(text)
        current = FILE_SECTIONS.index(section)
        for key, _value in root.value:
            if isinstance(key, ScalarNode) and key.value in FILE_SECTIONS:
                if FILE_SECTIONS.index(key.value) > current:
                    insertion_point = min(insertion_point, key.start_mark.index)
        index = insertion_point
        content = yaml.safe_dump({name: values}, sort_keys=False,
                                 allow_unicode=True, default_flow_style=False)
        replacement = ("" if not index or text[index - 1] == "\n" else "\n")
        replacement += section + ":\n" + "".join(
            "  " + line + "\n" for line in content.splitlines())
    return {"start": index, "end": index, "replacement": replacement,
            "line": text.count("\n", 0, index) + 1, "column": 1}


def remove_component(text: str, section: str, name: str) -> dict[str, Any]:
    """Remove an unambiguous declaration, only when no comments may be lost.

    The caller validates the resulting file with PESLite so that a component
    referenced by another device, event or connection cannot be removed.
    """
    if section not in COMPONENT_SECTIONS:
        raise WorkspaceError("unsupported component section")
    root = _mapping_document(text)
    target = _entry(root, section)
    if not isinstance(target, MappingNode) or target.flow_style:
        raise WorkspaceError("cannot remove a component from a non-block section")
    if len(target.value) <= 1:
        raise WorkspaceError("cannot remove the last entry of a section")
    matches = [(i, key, value) for i, (key, value) in enumerate(target.value)
               if isinstance(key, ScalarNode) and key.value == name]
    if len(matches) != 1:
        raise WorkspaceError("component is missing or ambiguous")
    position, key, _value = matches[0]
    if key.start_mark.column != 2:
        raise WorkspaceError("component declaration has ambiguous indentation")
    start = text.rfind("\n", 0, key.start_mark.index) + 1
    next_key = target.value[position + 1][0] if position + 1 < len(target.value) else None
    if next_key is not None:
        end = text.rfind("\n", 0, next_key.start_mark.index) + 1
    else:
        end = target.end_mark.index
        if target.end_mark.column:
            newline = text.find("\n", end)
            end = newline + 1 if newline >= 0 else len(text)
    if any("#" in line for line in text[start:end].splitlines()):
        raise WorkspaceError("cannot remove a component containing or preceding comments")
    return {"start": start, "end": end, "replacement": "", "line": key.start_mark.line + 1,
            "column": key.start_mark.column + 1}


class Workspace:
    """Directory-scoped, symlink-free access with optimistic versioned writes."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise WorkspaceError("workspace root must be a directory")
        self._lock = threading.RLock()

    def path(self, name: str, *, suffixes: frozenset[str] | set[str] | None = READ_SUFFIXES,
             directory: bool = False) -> Path:
        if directory and name == "":
            return self.root
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            raise WorkspaceError("a workspace-relative POSIX path is required")
        relative = PurePosixPath(name)
        if relative.is_absolute() or any(part in (".", "..") or part.startswith(".")
                                     for part in relative.parts):
            raise WorkspaceError("absolute, hidden or parent-directory paths are forbidden")
        candidate = self.root.joinpath(*relative.parts)
        # Reject links at every ancestor, including dangling links.  Resolving a
        # link within the root is also forbidden to avoid read/write aliasing.
        for segment in (candidate, *candidate.parents):
            if segment == self.root:
                break
            if segment.is_symlink():
                raise WorkspaceError("symlink paths are not allowed")
        if not candidate.resolve().is_relative_to(self.root):
            raise WorkspaceError("path escapes the workspace")
        if suffixes is not None and candidate.suffix.lower() not in suffixes:
            raise WorkspaceError("unsupported file type")
        return candidate

    def tree(self, directory: str = "") -> list[dict[str, Any]]:
        path = self.path(directory, suffixes=None, directory=True)
        if not path.is_dir():
            raise WorkspaceError("directory does not exist")
        return [{"name": child.name, "path": child.relative_to(self.root).as_posix(),
                 "directory": child.is_dir()}
                for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                if not child.name.startswith(".") and not child.is_symlink()
                and (child.is_dir() or child.suffix.lower() in READ_SUFFIXES)]

    def read(self, name: str) -> dict[str, str]:
        path = self.path(name, suffixes=TEXT_SUFFIXES)
        if not path.is_file() or path.stat().st_size > MAX_FILE_SIZE:
            raise WorkspaceError("text file not found or larger than 4 MiB")
        data = path.read_bytes()
        if len(data) > MAX_FILE_SIZE:
            raise WorkspaceError("text file larger than 4 MiB")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError("file is not UTF-8") from exc
        return {"path": name, "text": text, "version": content_version(data)}

    def write(self, name: str, text: str, version: str | None, *, create: bool = False) -> dict:
        path = self.path(name, suffixes=TEXT_SUFFIXES)
        if not isinstance(text, str):
            raise WorkspaceError("file content must be text")
        payload = text.encode("utf-8")
        if len(payload) > MAX_FILE_SIZE or not path.parent.is_dir():
            raise WorkspaceError("file too large or parent directory missing")
        with self._lock:
            if create:
                if path.exists():
                    raise FileConflict("file already exists")
            else:
                if not path.is_file():
                    raise WorkspaceError("file does not exist")
                if not version or content_version(path.read_bytes()) != version:
                    raise FileConflict("file changed on disk; reload before writing")
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".peslite-",
                                                 delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                if create:
                    # An atomic exclusive link avoids overwriting another creator.
                    os.link(temporary, path, follow_symlinks=False)
                else:
                    os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
                    if content_version(path.read_bytes()) != version:
                        raise FileConflict("file changed while saving")
                    os.replace(temporary, path)
                return {"path": name, "version": content_version(payload)}
            except FileExistsError as exc:
                raise FileConflict("file already exists") from exc
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def rename(self, source: str, destination: str, version: str) -> dict:
        old = self.path(source)
        new = self.path(destination)
        with self._lock:
            if not old.is_file() or not new.parent.is_dir():
                raise WorkspaceError("file or destination folder is missing")
            if new.exists():
                raise FileConflict("destination already exists")
            if content_version(old.read_bytes()) != version:
                raise FileConflict("file changed before rename")
            try:
                # Hard-link creation fails if another process creates the target.
                # Unlike Path.rename on POSIX, this never clobbers that file.
                os.link(old, new, follow_symlinks=False)
            except FileExistsError as exc:
                raise FileConflict("destination already exists") from exc
            old.unlink()
        return {"path": destination}

    def delete(self, name: str, version: str) -> dict:
        path = self.path(name)
        with self._lock:
            if not path.is_file():
                raise WorkspaceError("only ordinary files can be deleted")
            if content_version(path.read_bytes()) != version:
                raise FileConflict("file changed before deletion")
            path.unlink()
        return {"deleted": name}

    def csv(self, name: str, *, offset: int = 0, limit: int = 1000,
            stride: int = 1, byte_offset: int | None = None) -> dict[str, Any]:
        """Read at most MAX_SCANNED_ROWS records from a numeric CSV stream.

        A byte cursor avoids rescanning millions of rows during live tailing.
        The row-based offset is retained for existing clients. No incomplete
        record is consumed, so a growing simulator output can be read later.
        """
        path = self.path(name, suffixes={".csv"})
        if not path.is_file():
            raise WorkspaceError("CSV does not exist")
        if (offset < 0 or not 1 <= limit <= MAX_ROWS or not 1 <= stride <= 10000 or
                byte_offset is not None and byte_offset < 0):
            raise WorkspaceError("invalid CSV window")
        with path.open("rb") as handle:
            header = handle.readline(MAX_CSV_RECORD_BYTES + 1)
            if not header:
                return {"columns": [], "rows": [], "nextOffset": 0,
                        "nextByteOffset": 0, "eof": True}
            if len(header) > MAX_CSV_RECORD_BYTES or not header.endswith(b"\n"):
                raise WorkspaceError("CSV header exceeds the record limit or is incomplete")
            columns = next(csv.reader([header.decode("utf-8-sig")]))
            if not columns or any(not key for key in columns) or len(set(columns)) != len(columns):
                raise WorkspaceError("CSV header contains empty or duplicated column names")
            first_row = handle.tell()
            index = offset
            if byte_offset is not None:
                if byte_offset < first_row or byte_offset > path.stat().st_size:
                    raise FileConflict("CSV changed or truncated; restart reading from the beginning")
                handle.seek(byte_offset)
            else:
                index = 0
                for _ in range(offset):
                    record = handle.readline(MAX_CSV_RECORD_BYTES + 1)
                    if not record or not record.endswith(b"\n"):
                        break
                    index += 1

            rows: list[list[Any]] = []
            scanned = 0
            eof = False
            while scanned < MAX_SCANNED_ROWS and len(rows) < limit:
                beginning = handle.tell()
                record = handle.readline(MAX_CSV_RECORD_BYTES + 1)
                if not record:
                    eof = True
                    break
                if len(record) > MAX_CSV_RECORD_BYTES:
                    raise WorkspaceError("CSV record exceeds 1 MiB")
                if not record.endswith(b"\n"):
                    # In-progress writer; do not consume a partial final line.
                    handle.seek(beginning)
                    eof = True
                    break
                scanned += 1
                try:
                    cells = next(csv.reader([record.decode("utf-8")]))
                except (UnicodeDecodeError, csv.Error) as exc:
                    raise WorkspaceError(f"invalid CSV row: {exc}") from exc
                if len(cells) == len(columns) and index % stride == 0:
                    values: list[Any] = []
                    for cell in cells:
                        if not cell.strip():
                            values.append(None)
                        else:
                            try:
                                number = float(cell)
                                values.append(number if math.isfinite(number) else None)
                            except ValueError:
                                values.append(cell)
                    rows.append(values)
                index += 1
            if handle.tell() >= path.stat().st_size:
                eof = True
            return {"columns": columns, "rows": rows, "nextOffset": index,
                    "nextByteOffset": handle.tell(), "eof": eof}

    def summary(self, name: str) -> dict:
        path = self.path(name, suffixes={".json"})
        if path.name != "summary.json":
            raise WorkspaceError("only summary.json can be shown as a simulation summary")
        data = json.loads(self.read(name)["text"])
        model = path.with_name("simulation.pes")
        return {"summary": data,
                "simulationFile": model.relative_to(self.root).as_posix()
                if model.is_file() and not model.is_symlink() else None}

    def has_project_addons(self, name: str) -> bool:
        path = self.path(name, suffixes={".pes"})
        return any((directory / "PESaddons").is_dir() for directory in
                   [self.root, *[parent for parent in path.parents if parent != self.root
                                   and parent.is_relative_to(self.root)]])
