"""Read-only workspace access for the GUI host adapters."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

TEXT_SUFFIXES = frozenset({".pes", ".pesdbg", ".yaml", ".yml", ".json"})
READ_SUFFIXES = TEXT_SUFFIXES | {".csv"}
MAX_FILE_SIZE = 4 * 1024 * 1024
MAX_ROWS = 4000
MAX_SCANNED_ROWS = 20000
MAX_CSV_RECORD_BYTES = 1024 * 1024


class WorkspaceError(ValueError):
    """A rejected client operation; HTTP adapters use its status code."""

    status = 400


class FileConflict(WorkspaceError):
    """A streamed result cursor no longer refers to the current file."""

    status = 409


def content_version(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Workspace:
    """Directory-scoped, symlink-free access to source and result files."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise WorkspaceError("workspace root must be a directory")

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
