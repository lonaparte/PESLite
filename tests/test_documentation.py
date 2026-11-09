"""Keep duplicated public documentation consistent."""

from pathlib import Path

import pytest


pytestmark = pytest.mark.static

ROOT = Path(__file__).resolve().parents[1]


def _bullet_list_after(path: Path, marker: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    _, separator, remainder = text.partition(marker)
    assert separator, f"{marker!r} is missing from {path.relative_to(ROOT)}"

    items = []
    for line in remainder.splitlines():
        if line.startswith("- "):
            items.append(line[2:])
        elif items and line.strip():
            break
    assert items, f"no list follows {marker!r} in {path.relative_to(ROOT)}"
    return items


def test_homepage_support_list_matches_readme_features():
    readme_features = _bullet_list_after(ROOT / "README.md", "## Features")
    homepage_features = _bullet_list_after(ROOT / "docs/index.md", "PESLite supports:")
    assert homepage_features == readme_features
