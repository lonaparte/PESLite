"""Validate the built PESLite wheel matrix and its disjoint file ownership."""

from __future__ import annotations

import json
import sys
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

MANIFEST = Path(__file__).resolve().parents[1] / "packages/distributions.json"
EXPECTED = set(json.loads(MANIFEST.read_text(encoding="utf-8"))["distributions"])


def _installed_path(name: str) -> str:
    """Return the destination represented by a path inside a wheel."""
    if ".data/" not in name:
        return name
    _distribution, relative = name.split(".data/", 1)
    scheme, separator, path = relative.partition("/")
    if not separator:
        return name
    return path if scheme in {"data", "purelib", "platlib"} else f"{scheme}/{path}"


def _wheel(path: Path) -> tuple[str, set[str]]:
    with ZipFile(path) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
    owned = {_installed_path(name) for name in names if ".dist-info/" not in name}
    return metadata["Name"], owned


def main(directory: str = "dist") -> int:
    root = Path(directory)
    wheels = dict(_wheel(path) for path in sorted(root.glob("*/*.whl")))
    if set(wheels) != EXPECTED:
        raise RuntimeError(f"wheel set differs: {sorted(wheels)}")
    if len(list(root.glob("*/*.tar.gz"))) != len(EXPECTED):
        raise RuntimeError("each distribution must have one source archive")
    for distribution in EXPECTED:
        artifacts = list((root / distribution).iterdir())
        if len(artifacts) != 2:
            raise RuntimeError(f"{distribution} must have one wheel and one source archive")

    code = tuple(EXPECTED)
    for index, left in enumerate(code):
        for right in code[index + 1:]:
            overlap = wheels[left] & wheels[right]
            if overlap:
                raise RuntimeError(f"{left} and {right} both own {sorted(overlap)}")

    required = {
        "peslite-core": (
            "peslite/addons/functions/__init__.py",
            "share/peslite/examples/templates/template.pes",
        ),
        "peslite": (
            "peslite/addons/components/magcomponent.py",
            "peslite/addons/components/thermalcomponent.py",
            "peslite/addons/components/thermalnetwork.py",
            "peslite/addons/functions/legacy_yaml.py",
            "peslite/addons/functions/thermal/__init__.py",
            "peslite/addons/functions/stability.py",
            "share/peslite/examples/gfl-example.pes",
            "share/peslite/examples/debug/gfl-debug-example.pesdbg",
        ),
        "peslite-gui": (
            "peslite/gui/legacy_yaml.py",
            "peslite/gui/host.py",
            "peslite/gui/service.py",
            "peslite/gui/web_service.py",
            "peslite/gui/workspace.py",
            "peslite/gui/preview/dist/pes-preview.es.js",
            "peslite/gui/preview/dist/pes-preview.iife.js",
            "peslite/gui/preview/demo.html",
            "peslite/gui/preview/workbench.css",
            "peslite/gui/preview/workbench.js",
            "peslite/gui/web_dist/index.html",
        ),
        "peslite-full": (
            "peslite/addons/components/magport.py",
            "peslite/addons/components/thermalport.py",
            "peslite/addons/components/magcircuit.py",
            "peslite/addons/functions/magport.py",
            "peslite/addons/functions/magcircuit.py",
        ),
    }
    for distribution, paths in required.items():
        for path in paths:
            if path not in wheels[distribution]:
                raise RuntimeError(f"{distribution} does not own {path}")
    print("validated distributions:", ", ".join(sorted(wheels)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
