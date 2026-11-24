"""The installable distributions compose without overlapping owned files."""

import json
import re
from pathlib import Path

import pytest
import tomllib
import yaml
from packages.distributions import selections

pytestmark = pytest.mark.static

ROOT = Path(__file__).resolve().parents[1]
ADDON_MANIFEST = ROOT / "src/peslite/addons/addons.toml"
ADDON_DIRECTORIES = {"component": "components", "control": "control", "function": "functions"}
PROJECTS = {
    "peslite-core": ROOT / "pyproject.toml",
    "peslite": ROOT / "packages/peslite/pyproject.toml",
    "peslite-gui": ROOT / "packages/gui/pyproject.toml",
    "peslite-full": ROOT / "packages/full/pyproject.toml",
}


def _project(path):
    with path.open("rb") as handle:
        return tomllib.load(handle)["project"]


def _requirements(project):
    return {re.split(r"[<>=!~; ]", requirement, maxsplit=1)[0]
            for requirement in project.get("dependencies", ())}


def _addons():
    with ADDON_MANIFEST.open("rb") as stream:
        return tomllib.load(stream)


def test_distribution_names_versions_and_dependencies_form_the_install_matrix():
    projects = {name: _project(path) for name, path in PROJECTS.items()}
    assert set(ROOT.rglob("pyproject.toml")) == set(PROJECTS.values())
    assert {project["name"] for project in projects.values()} == set(PROJECTS)
    versions = {project["version"] for project in projects.values()}
    assert len(versions) == 1
    version, = versions
    npm_projects = [
        ROOT / "src/peslite/gui/preview/package.json",
        ROOT / "src/peslite/gui/ui/package.json",
        ROOT / "src/peslite/gui/vscode/package.json",
    ]
    assert {json.loads(path.read_text(encoding="utf-8"))["version"]
            for path in npm_projects} == {version}
    assert _requirements(projects["peslite-core"]) == {"numpy", "scipy", "PyYAML"}
    assert _requirements(projects["peslite"]) == {"peslite-core", "matplotlib"}
    assert _requirements(projects["peslite-gui"]) == {"peslite"}
    assert _requirements(projects["peslite-full"]) == {
        "peslite-gui", "onnx", "onnxruntime"}
    assert f"peslite-core=={version}" in projects["peslite"]["dependencies"]
    assert f"peslite=={version}" in projects["peslite-gui"]["dependencies"]
    assert f"peslite-gui=={version}" in projects["peslite-full"]["dependencies"]


def test_code_distributions_own_disjoint_source_files():
    owned = selections()
    code = tuple(owned)
    for index, left in enumerate(code):
        for right in code[index + 1:]:
            assert owned[left].isdisjoint(owned[right])
    assert Path("src/peslite/addons/addons.toml") in owned["peslite-core"]
    for name, addon in _addons().items():
        owner = "peslite" if addon["installation"] == "standard" else "peslite-full"
        for category, directory in ADDON_DIRECTORIES.items():
            if not addon[category]:
                continue
            path = Path(f"src/peslite/addons/{directory}/{name}.py")
            assert path in owned[owner]
            assert all(path not in files for distribution, files in owned.items()
                       if distribution != owner)
    assert Path("src/peslite/addons/components/coupled_coils.py") not in set().union(
        *owned.values()
    )
    assert Path("src/peslite/components/cable.py") in owned["peslite-core"]
    assert Path("src/peslite/gui/legacy_yaml.py") in owned["peslite-gui"]
    assert Path("src/peslite/gui/service.py") in owned["peslite-gui"]
    assert Path("src/peslite/gui/preview/parser.ts") in owned["peslite-gui"]
    assert Path("src/peslite/gui/preview/package-lock.json") in owned["peslite-gui"]
    assert Path("src/peslite/gui/ui/src/App.tsx") in owned["peslite-gui"]
    assert Path("src/peslite/gui/ui/package-lock.json") in owned["peslite-gui"]
    assert Path("src/peslite/gui/vscode/src/extension.ts") in owned["peslite-gui"]
    assert Path("src/peslite/gui/vscode/package-lock.json") in owned["peslite-gui"]


def test_gui_preview_uses_the_declarative_distribution_build():
    with (ROOT / "packages/distributions.json").open(encoding="utf-8") as handle:
        gui = json.load(handle)["distributions"]["peslite-gui"]
    steps = gui["build-steps"]
    assert [(step["cwd"], step["command"][:2]) for step in steps] == [
        ("src/peslite/gui/preview", ["npm", "ci"]),
        ("src/peslite/gui/preview", ["npm", "run"]),
        ("src/peslite/gui/ui", ["npm", "ci"]),
        ("src/peslite/gui/ui", ["npm", "run"]),
    ]


def test_templates_and_cases_have_distinct_owners():
    with (ROOT / "packages/distributions.json").open(encoding="utf-8") as handle:
        distributions = json.load(handle)["distributions"]
    assert distributions["peslite-core"]["files"] == ["examples/templates/*"]
    assert distributions["peslite"]["files"] == [
        "examples/*.pes", "examples/debug/*.pesdbg",
    ]


def test_public_documentation_describes_the_distribution_matrix_and_gui():
    installation = (ROOT / "docs/Getting-Started.md").read_text(encoding="utf-8")
    for distribution in PROJECTS:
        assert distribution in installation
    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Graphical Interface: Graphical-Interface.md" in navigation


def test_addon_documentation_tracks_the_manifest():
    rows = {}
    for line in (ROOT / "docs/List-of-Addons.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[cells[0].strip("`")] = {
            "version": cells[1],
            "categories": tuple(item.strip().strip("`") for item in cells[2].split(",")),
            "installation": cells[3],
            "dependencies": (() if cells[4] == "—" else
                             tuple(item.strip().strip("`") for item in cells[4].split(","))),
        }
    addons = _addons()
    assert set(rows) == set(addons)
    assert rows == {
        name: {
            "version": addon["version"],
            "categories": tuple(category for category in ADDON_DIRECTORIES if addon[category]),
            "installation": addon["installation"],
            "dependencies": tuple(addon["dependencies"]),
        }
        for name, addon in addons.items()
    }


def test_release_uses_one_approval_gate_for_the_distribution_matrix():
    with (ROOT / ".github/workflows/release.yml").open(encoding="utf-8") as stream:
        jobs = yaml.safe_load(stream)["jobs"]

    assert jobs["approve-release"]["environment"]["name"] == "release"
    publish = jobs["publish"]
    needs = publish["needs"] if isinstance(publish["needs"], list) else [publish["needs"]]
    assert "approve-release" in needs
    assert publish["environment"]["name"] == "${{ matrix.environment }}"
    expected = {
        "peslite": "pypi",
        "peslite-core": "pypi",
        "peslite-gui": "pypi-gui",
        "peslite-full": "pypi-full",
    }
    matrix = publish["strategy"]["matrix"]["include"]
    assert {entry["distribution"]: entry["environment"] for entry in matrix} == expected
    publisher, = [step for step in publish["steps"]
                  if step.get("uses") == "pypa/gh-action-pypi-publish@release/v1"]
    assert publisher["with"]["packages-dir"] == "dist/${{ matrix.distribution }}"
