"""Temporary block positions for one GUI preview session."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

FORMAT = "peslite.preview-layout"
MAX_OFFSET = 1e6


def empty_layout() -> dict[str, Any]:
    return {"format": FORMAT, "version": 1, "views": {}}


def normalize_layout(value: Any) -> dict[str, Any]:
    """Validate the transport object shared with the browser renderer."""
    if not isinstance(value, dict):
        raise ValueError("a preview layout must be an object")
    if value.get("format", FORMAT) != FORMAT or value.get("version", 1) != 1:
        raise ValueError("unsupported preview layout format or version")
    if not isinstance(value.get("views", {}), dict):
        raise ValueError("preview layout views must be an object")
    result = empty_layout()
    for view, nodes in value.get("views", {}).items():
        if not isinstance(view, str) or not view or not isinstance(nodes, dict):
            continue
        clean = {}
        for node, offset in nodes.items():
            if not isinstance(node, str) or not node or not isinstance(offset, dict):
                continue
            dx, dy = offset.get("dx"), offset.get("dy")
            valid = all(isinstance(item, (int, float)) and not isinstance(item, bool)
                        and math.isfinite(item) and abs(item) <= MAX_OFFSET for item in (dx, dy))
            if valid and (dx or dy):
                clean[node] = {"dx": dx, "dy": dy}
        if clean:
            result["views"][view] = clean
    return result


class PreviewLayoutCache:
    """A session cache that removes its backing file when cleared or closed."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.layouts: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def load(self, key: str) -> dict[str, Any]:
        with self._lock:
            return self.layouts.get(key, empty_layout())

    def save(self, key: str, layout: Any) -> dict[str, Any]:
        clean = normalize_layout(layout)
        with self._lock:
            self.layouts[key] = clean
            if self.path is None:
                descriptor, name = tempfile.mkstemp(prefix="peslite-preview-", suffix=".json")
                os.close(descriptor)
                self.path = Path(name)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=self.path.stem, suffix=".tmp",
                                                 delete=False) as handle:
                    json.dump(self.layouts, handle, sort_keys=True)
                    temporary = Path(handle.name)
                os.replace(temporary, self.path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return clean

    def clear(self) -> None:
        with self._lock:
            self.layouts.clear()
            if self.path is not None:
                self.path.unlink(missing_ok=True)
                self.path = None

    close = clear
