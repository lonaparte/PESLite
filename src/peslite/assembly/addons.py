"""Read, validate and assemble installed PESLite addons."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import import_module, invalidate_caches
from pathlib import Path
from pkgutil import iter_modules
from types import MappingProxyType, ModuleType
from typing import Callable, Iterable, Mapping
import tomllib


_CATEGORIES = {"control": "control", "component": "components", "function": "functions"}
_INSTALLATIONS = frozenset(("standard", "full"))
_REQUIRED_FIELDS = frozenset(("version", "installation", "dependencies", "summary", *_CATEGORIES))
_FIELDS = _REQUIRED_FIELDS | {"simulation_reader"}


@dataclass(frozen=True)
class AddonSpec:
    """One addon and the categories in which it provides a module."""

    name: str
    version: str
    installation: str
    dependencies: tuple[str, ...]
    summary: str
    control: bool = False
    component: bool = False
    function: bool = False
    simulation_reader: bool = False

    def module(self, category: str) -> str | None:
        """Return the convention-based module name for one present category."""
        return self.name if getattr(self, category) else None


def _identifier(value, where: str) -> str:
    if not isinstance(value, str) or not value.isidentifier() or value.startswith("_"):
        raise RuntimeError(f"{where} must be a public Python identifier")
    return value


def _version(value, where: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{where} must use major.minor.patch")
    parts = value.split(".")
    if len(parts) != 3 or any(not part.isdecimal() for part in parts):
        raise RuntimeError(f"{where} must use major.minor.patch")
    return value


def _dependency_order(specs: Mapping[str, AddonSpec]) -> tuple[str, ...]:
    """Return one deterministic dependency-first order and reject cycles."""
    ordered: list[str] = []
    active: set[str] = set()
    complete: set[str] = set()

    def visit(name: str) -> None:
        if name in complete:
            return
        if name in active:
            raise RuntimeError(f"addon dependency cycle includes {name!r}")
        active.add(name)
        for dependency in specs[name].dependencies:
            visit(dependency)
        active.remove(name)
        complete.add(name)
        ordered.append(name)

    for name in specs:
        visit(name)
    return tuple(ordered)


def _load() -> tuple[Mapping[str, AddonSpec], tuple[str, ...]]:
    path = Path(__file__).parents[1] / "addons" / "addons.toml"
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    specs: dict[str, AddonSpec] = {}
    for raw_name, raw in data.items():
        name = _identifier(raw_name, "addon name")
        if not isinstance(raw, dict):
            raise RuntimeError(f"{name} must be an addon table")
        missing = _REQUIRED_FIELDS - set(raw)
        unknown = set(raw) - _FIELDS
        if missing:
            raise RuntimeError(f"{name} is missing fields {sorted(missing)}")
        if unknown:
            raise RuntimeError(f"{name} has unknown fields {sorted(unknown)}")
        categories = {}
        for category in _CATEGORIES:
            present = raw[category]
            if not isinstance(present, bool):
                raise RuntimeError(f"{name}.{category} must be true or false")
            categories[category] = present
        simulation_reader = raw.get("simulation_reader", False)
        if not isinstance(simulation_reader, bool):
            raise RuntimeError(f"{name}.simulation_reader must be true or false")
        if simulation_reader and not categories["function"]:
            raise RuntimeError(f"{name}.simulation_reader requires a function module")
        if not any(categories.values()):
            raise RuntimeError(f"{name} must declare at least one addon category")
        installation = raw["installation"]
        if installation not in _INSTALLATIONS:
            raise RuntimeError(f"{name}.installation must be standard or full")
        summary = raw["summary"]
        if not isinstance(summary, str) or not summary.strip():
            raise RuntimeError(f"{name}.summary must be nonempty")
        dependencies = raw["dependencies"]
        if not isinstance(dependencies, list) or len(set(dependencies)) != len(dependencies):
            raise RuntimeError(f"{name}.dependencies must be a list without duplicates")
        specs[name] = AddonSpec(
            name=name,
            version=_version(raw["version"], f"{name}.version"),
            installation=installation,
            dependencies=tuple(
                _identifier(item, f"{name}.dependencies") for item in dependencies
            ),
            summary=summary.strip(),
            simulation_reader=simulation_reader,
            **categories,
        )
    for spec in specs.values():
        missing = set(spec.dependencies) - specs.keys()
        if missing:
            raise RuntimeError(f"{spec.name} depends on unknown addons {sorted(missing)}")
        if spec.installation == "standard":
            full = [name for name in spec.dependencies if specs[name].installation == "full"]
            if full:
                raise RuntimeError(f"standard addon {spec.name} depends on full addons {full}")
    frozen = MappingProxyType(specs)
    return frozen, _dependency_order(frozen)


ADDONS, _ORDER = _load()


class AddonUnavailableError(ImportError):
    """An addon module declared by the manifest is not installed."""


def import_addon(name: str, category: str) -> ModuleType:
    """Import one declared addon module through its category convention."""
    try:
        spec = ADDONS[name]
        package = _CATEGORIES[category]
    except KeyError as exc:
        raise KeyError(f"unknown addon or category: {exc.args[0]}") from None
    module = spec.module(category)
    if module is None:
        raise ValueError(f"addon {name!r} has no {category} module")
    qualified = f"peslite.addons.{package}.{module}"
    try:
        return import_module(qualified)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == qualified or qualified.startswith(exc.name + ".")):
            raise AddonUnavailableError(
                f"addon {name!r} is unavailable in this installation"
            ) from None
        raise


def import_category(category: str) -> tuple[ModuleType, ...]:
    """Import installed modules of one category in dependency order."""
    _CATEGORIES[category]
    imported = []
    for name in _ORDER:
        module = ADDONS[name].module(category)
        if module is None:
            continue
        try:
            imported.append(import_addon(name, category))
        except AddonUnavailableError:
            continue
    return tuple(imported)


def simulation_readers() -> tuple[Callable[[str, Path], Mapping[str, object] | None], ...]:
    """Return installed simulation readers declared by function addons."""
    readers = []
    for name in _ORDER:
        if not ADDONS[name].simulation_reader:
            continue
        try:
            module = import_addon(name, "function")
        except AddonUnavailableError:
            continue
        reader = getattr(module, "read_simulation", None)
        if not callable(reader):
            raise RuntimeError(f"simulation-reader addon {name!r} must define read_simulation")
        readers.append(reader)
    return tuple(readers)


def _import_modules(package: str, paths: Iterable[str]) -> tuple[ModuleType, ...]:
    """Import public project-local modules from one addon category."""
    prefix = f"{package}."
    names = sorted(
        info.name for info in iter_modules(paths, prefix)
        if not info.name.rsplit(".", 1)[-1].startswith("_")
    )
    return tuple(import_module(name) for name in names)


@cache
def register_installed() -> None:
    """Register installed control and component addons."""
    import_category("control")
    import_category("component")


def _merge_path(package: ModuleType, directory: Path) -> bool:
    """Append one real directory to a package search path, once."""
    if not directory.is_dir():
        return False
    resolved = str(directory.resolve())
    if resolved not in package.__path__:
        package.__path__.append(resolved)
    return True


def register_project(simulation_file: str | Path) -> tuple[ModuleType, ...]:
    """Register project-local addons beside one simulation file."""
    root = Path(simulation_file).resolve().parent / "PESaddons"
    if not root.is_dir():
        return ()

    addons = import_module("peslite.addons")
    invalidate_caches()
    _merge_path(addons, root)

    locations = []
    for name in ("control", "components", "functions"):
        directory = root / name
        if not directory.is_dir():
            continue
        package = import_module(f"peslite.addons.{name}")
        _merge_path(package, directory)
        if name != "functions":
            locations.append((package, directory))

    imported = []
    for package, directory in locations:
        imported.extend(_import_modules(package.__name__, (str(directory),)))
    return tuple(imported)
