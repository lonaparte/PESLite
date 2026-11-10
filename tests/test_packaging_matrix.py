"""The installable distributions compose without overlapping owned files."""

import json
import re
from pathlib import Path

import pytest
import yaml
from scripts.distributions import selections

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

pytestmark = pytest.mark.static

ROOT = Path(__file__).resolve().parents[1]
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


def test_distribution_names_versions_and_dependencies_form_the_install_matrix():
    projects = {name: _project(path) for name, path in PROJECTS.items()}
    assert set(ROOT.rglob("pyproject.toml")) == set(PROJECTS.values())
    assert {project["name"] for project in projects.values()} == set(PROJECTS)
    versions = {project["version"] for project in projects.values()}
    assert len(versions) == 1
    version, = versions
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
    for left, right in (("peslite-core", "peslite"),
                        ("peslite-core", "peslite-gui"),
                        ("peslite", "peslite-gui")):
        assert owned[left].isdisjoint(owned[right])
    assert Path("src/peslite/addons/__init__.py") in owned["peslite-core"]
    assert Path("src/peslite/addons/functions/__init__.py") in owned["peslite-core"]
    assert Path("src/peslite/addons/components/cable.py") in owned["peslite"]
    assert Path("src/peslite/addons/functions/small_signal.py") in owned["peslite"]
    assert Path("src/peslite/addons/control/mpc.py") in owned["peslite"]
    assert Path("src/peslite/addons/control/predictive.py") in owned["peslite"]
    assert Path("src/peslite/addons/control/_native_mpc.py") in owned["peslite"]
    assert Path("src/peslite/gui/service.py") in owned["peslite-gui"]
    assert Path("src/peslite/gui/preview/parser.ts") in owned["peslite-gui"]
    assert Path("src/peslite/gui/preview/package-lock.json") in owned["peslite-gui"]


def test_gui_preview_uses_the_declarative_distribution_build():
    with (ROOT / "packages/distributions.json").open(encoding="utf-8") as handle:
        gui = json.load(handle)["distributions"]["peslite-gui"]
    steps = gui["build-steps"]
    assert [step["command"][:2] for step in steps] == [["npm", "ci"], ["npm", "run"]]
    assert {step["cwd"] for step in steps} == {"src/peslite/gui/preview"}


def test_templates_and_cases_have_distinct_owners():
    with (ROOT / "packages/distributions.json").open(encoding="utf-8") as handle:
        distributions = json.load(handle)["distributions"]
    assert distributions["peslite-core"]["files"] == ["examples/templates/*"]
    assert distributions["peslite"]["files"] == ["examples/*.pes"]


def test_public_documentation_describes_the_distribution_matrix_and_gui():
    installation = (ROOT / "docs/Getting-Started.md").read_text(encoding="utf-8")
    for distribution in PROJECTS:
        assert distribution in installation
    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Graphical Interface: Graphical-Interface.md" in navigation


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
