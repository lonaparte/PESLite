"""Stage and build the PESLite distributions from the unified source tree."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "packages/distributions.json"
SUPPORT_FILES = ("README.md", "LICENSE")


def _matching_files(pattern: str) -> set[Path]:
    files = {path.relative_to(ROOT) for path in ROOT.glob(pattern) if path.is_file()}
    if not files:
        raise RuntimeError(f"distribution pattern matches no files: {pattern}")
    return files


def _configured_files(config: dict) -> set[Path]:
    return set().union(*(_matching_files(pattern) for pattern in config.get("files", ())))


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
    if not groups["addons"].isdisjoint(groups["gui"]):
        raise RuntimeError("addon and GUI file groups overlap")
    if not (groups["addons"] | groups["gui"]) <= groups["all"]:
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
    code = ("peslite-core", "peslite", "peslite-gui")
    for index, left in enumerate(code):
        for right in code[index + 1:]:
            overlap = selected[left] & selected[right]
            if overlap:
                raise RuntimeError(f"{left} and {right} both own {sorted(map(str, overlap))}")
    if set().union(*(selected[name] for name in code)) != groups["all"]:
        raise RuntimeError("the code distributions do not own every Python source file exactly once")
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
            command = [sys.executable, "-m", "build", "--outdir", str(package_output)]
            if not isolation:
                command.append("--no-isolation")
            command.append(str(project))
            subprocess.run(command, check=True)


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
    args = parser.parse_args(argv)
    if args.command == "stage":
        stage(args.name, args.destination)
    elif args.command == "build":
        build(args.output, isolation=not args.no_isolation)
    else:
        selections()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
