"""Stage and build the PESLite distributions from the unified source tree."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "packages/distributions.json"
SUPPORT_FILES = ("README.md", "LICENSE")
ADDON_MANIFEST = ROOT / "src/peslite/addons/addons.toml"
ADDON_DIRECTORIES = {"control": "control", "component": "components", "function": "functions"}


def _matching_files(pattern: str) -> set[Path]:
    files = {path.relative_to(ROOT) for path in ROOT.glob(pattern) if path.is_file()}
    if not files:
        raise RuntimeError(f"distribution pattern matches no files: {pattern}")
    return files


def _configured_files(config: dict) -> set[Path]:
    return set().union(*(_matching_files(pattern) for pattern in config.get("files", ())))


def _addon_groups() -> dict[str, set[Path]]:
    """Assign every public addon module from the authoritative manifest."""
    with ADDON_MANIFEST.open("rb") as stream:
        data = tomllib.load(stream)
    groups = {"addons": set(), "full-addons": set()}
    declared = set()
    for name, addon in data.items():
        group = "addons" if addon["installation"] == "standard" else "full-addons"
        for field, directory in ADDON_DIRECTORIES.items():
            if not addon[field]:
                continue
            path = Path(f"src/peslite/addons/{directory}/{name}.py")
            if not (ROOT / path).is_file():
                raise RuntimeError(f"addon {name!r} is missing {path}")
            if path in declared:
                raise RuntimeError(f"addon module is owned twice: {path}")
            declared.add(path)
            groups[group].add(path)
    actual = {
        path.relative_to(ROOT)
        for directory in ADDON_DIRECTORIES.values()
        for path in (ROOT / "src/peslite/addons" / directory).glob("*.py")
        if path.stem != "__init__" and not path.stem.startswith("_")
    }
    if actual != declared:
        raise RuntimeError(
            f"addon manifest differs from implementation files: missing "
            f"{sorted(map(str, actual - declared))}, absent {sorted(map(str, declared - actual))}"
        )
    return groups


def _manifest() -> tuple[dict[str, set[Path]], dict]:
    with MANIFEST.open(encoding="utf-8") as handle:
        data = json.load(handle)
    groups = {}
    for name, patterns in data["file-groups"].items():
        included, excluded = set(), set()
        for pattern in patterns:
            destination = excluded if pattern.startswith("!") else included
            destination.update(_matching_files(pattern.lstrip("!")))
        groups[name] = included - excluded
    groups.update(_addon_groups())
    if not groups["addons"].isdisjoint(groups["gui"]):
        raise RuntimeError("addon and GUI file groups overlap")
    if not (groups["addons"] | groups["full-addons"] | groups["gui"]) <= groups["all"]:
        raise RuntimeError("every optional file must belong to the complete source group")
    return groups, data["distributions"]


def selections() -> dict[str, set[Path]]:
    """Return and validate the files owned by every code distribution."""
    groups, distributions = _manifest()
    selected = {}
    for name, config in distributions.items():
        files = set().union(*(groups[group] for group in config.get("include", ())))
        files.difference_update(*(groups[group] for group in config.get("exclude", ())))
        selected[name] = files
    code = tuple(distributions)
    for index, left in enumerate(code):
        for right in code[index + 1:]:
            overlap = selected[left] & selected[right]
            if overlap:
                raise RuntimeError(f"{left} and {right} both own {sorted(map(str, overlap))}")
    if set().union(*(selected[name] for name in code)) != groups["all"]:
        raise RuntimeError("the code distributions do not own every source file exactly once")
    staged = {name: selected[name] | _configured_files(config)
              for name, config in distributions.items()}
    names = tuple(distributions)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = staged[left] & staged[right]
            if overlap:
                raise RuntimeError(f"{left} and {right} both stage {sorted(map(str, overlap))}")
    return selected


def stage(name: str, destination: str | Path) -> Path:
    """Create one self-contained build tree without changing the source checkout."""
    _groups, distributions = _manifest()
    selected = selections()
    if name not in distributions:
        raise KeyError(f"unknown distribution {name!r}; expected {sorted(distributions)}")
    target = Path(destination).resolve()
    if target.exists():
        if any(target.iterdir()):
            raise FileExistsError(f"staging directory is not empty: {target}")
    else:
        target.mkdir(parents=True)
    config = distributions[name]
    shutil.copy2(ROOT / config["project"], target / "pyproject.toml")
    for support in SUPPORT_FILES:
        shutil.copy2(ROOT / support, target / support)
    for relative in sorted(selected[name]):
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, output)
    for relative in sorted(_configured_files(config)):
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, output)
    return target


def _run_build_steps(project: Path, config: dict) -> None:
    """Run declarative asset builds inside a staged distribution tree."""
    for step in config.get("build-steps", ()):
        directory = (project / step["cwd"]).resolve()
        if not directory.is_relative_to(project) or not directory.is_dir():
            raise RuntimeError(f"invalid distribution build directory: {step['cwd']}")
        command = step.get("command")
        if not isinstance(command, list) or not command or not all(
                isinstance(part, str) and part for part in command):
            raise RuntimeError(f"invalid distribution build command: {command!r}")
        subprocess.run(command, cwd=directory, check=True)


def build(output: str | Path, *, isolation: bool = True) -> None:
    """Build every declared wheel and source archive sequentially."""
    _groups, distributions = _manifest()
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name in distributions:
        package_output = destination / name
        package_output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"peslite-{name}-") as temporary:
            project = stage(name, Path(temporary) / name)
            _run_build_steps(project, distributions[name])
            command = [sys.executable, "-m", "build", "--outdir", str(package_output)]
            if not isolation:
                command.append("--no-isolation")
            command.append(str(project))
            subprocess.run(command, check=True)


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


def check(directory: str | Path = "dist") -> None:
    """Validate the built wheel matrix and its disjoint installed ownership."""
    _groups, distributions = _manifest()
    expected = set(distributions)
    root = Path(directory)
    wheels = dict(_wheel(path) for path in sorted(root.glob("*/*.whl")))
    if set(wheels) != expected:
        raise RuntimeError(f"wheel set differs: {sorted(wheels)}")
    if len(list(root.glob("*/*.tar.gz"))) != len(expected):
        raise RuntimeError("each distribution must have one source archive")
    for distribution in expected:
        artifacts = list((root / distribution).iterdir())
        if len(artifacts) != 2:
            raise RuntimeError(f"{distribution} must have one wheel and one source archive")

    names = tuple(expected)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = wheels[left] & wheels[right]
            if overlap:
                raise RuntimeError(f"{left} and {right} both own {sorted(overlap)}")

    required: dict[str, list[str]] = {
        "peslite-core": [
            "peslite/addons/addons.toml",
            "share/peslite/examples/templates/template.pes",
        ],
        "peslite": [
            "share/peslite/examples/gfl-example.pes",
            "share/peslite/examples/debug/gfl-debug-example.pesdbg",
        ],
        "peslite-gui": [
            "peslite/gui/legacy_yaml.py",
            "peslite/gui/host.py",
            "peslite/gui/service.py",
            "peslite/gui/web_service.py",
            "peslite/gui/workspace.py",
            "peslite/gui/preview/dist/pes-preview.es.js",
            "peslite/gui/preview/dist/pes-preview.iife.js",
            "peslite/gui/web_dist/index.html",
        ],
        "peslite-full": [],
    }
    with ADDON_MANIFEST.open("rb") as stream:
        addons = tomllib.load(stream)
    for name, addon in addons.items():
        distribution = "peslite" if addon["installation"] == "standard" else "peslite-full"
        required[distribution].extend(
            f"peslite/addons/{directory}/{name}.py"
            for field, directory in ADDON_DIRECTORIES.items()
            if addon[field]
        )
    for distribution, paths in required.items():
        for path in paths:
            if path not in wheels[distribution]:
                raise RuntimeError(f"{distribution} does not own {path}")
    print("validated distributions:", ", ".join(sorted(wheels)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage_parser = commands.add_parser("stage", help="create one temporary build tree")
    stage_parser.add_argument("name")
    stage_parser.add_argument("destination")
    build_parser = commands.add_parser("build", help="build all distributions")
    build_parser.add_argument("output")
    build_parser.add_argument("--no-isolation", action="store_true")
    commands.add_parser("validate", help="validate the distribution file ownership")
    check_parser = commands.add_parser("check", help="validate built distribution artifacts")
    check_parser.add_argument("directory", nargs="?", default="dist")
    args = parser.parse_args(argv)
    if args.command == "stage":
        stage(args.name, args.destination)
    elif args.command == "build":
        build(args.output, isolation=not args.no_isolation)
    elif args.command == "check":
        check(args.directory)
    else:
        selections()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
