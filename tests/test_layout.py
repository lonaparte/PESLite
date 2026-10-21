"""The package has five code parts; the root examples directory contains YAML-format .pes files."""

import ast
from pathlib import Path

import peslite

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "peslite"
PARTS = ("solver", "control", "components", "assembly", "addons")

# What each part may import. The numerical solver kernel is at the bottom; the application runner
# (solver.simulation) is deliberately at the top. Optional analysis addons consume observations,
# completed simulations and shared solver primitives; core execution does not depend on them.
BELOW = {
    "solver": {"solver"},
    "control": {"solver", "control"},
    "components": {"solver", "control", "components"},
    "assembly": {"solver", "control", "components", "assembly"},
    "addons": {"solver", "control", "components", "addons"},
}
TOP = {"solver/simulation.py"}


def _imports(path):
    """Return the package parts imported through relative imports by a path."""
    rel = path.relative_to(PACKAGE).with_suffix("")
    here = list(rel.parts[:-1])
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = here[:len(here) - (node.level - 1)]
            target = base + (node.module.split(".") if node.module else [])
            if target:
                out.add(target[0])
    return out


def test_the_package_has_five_code_parts():
    entries = {p.name for p in PACKAGE.iterdir() if p.name != "__pycache__"}
    assert entries == {"__init__.py", *PARTS}


def test_the_root_examples_directory_contains_only_pes_files():
    examples = ROOT / "examples"
    assert examples.is_dir()
    assert {path.suffix for path in examples.iterdir()} == {".pes"}


def test_each_part_imports_only_the_parts_below_it():
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE)
        if rel.as_posix() in ("__init__.py", *TOP):
            continue
        wrong = _imports(path) - BELOW[rel.parts[0]]
        assert not wrong, f"{rel} imports {sorted(wrong)}"


def test_the_package_is_imported_from_src():
    assert Path(peslite.__file__).resolve() == PACKAGE / "__init__.py"


def test_the_command_line_finds_bundled_examples_by_name():
    from peslite.solver.simulation import _config_path

    found = _config_path("gfl-example")
    assert found.parent == ROOT / "examples"
    assert found.stem == "gfl-example"
    assert _config_path(None) == found
