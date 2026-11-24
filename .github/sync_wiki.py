#!/usr/bin/env python3
"""Render ``docs/`` as a GitHub Wiki checkout.

The main repository is the only documentation source. This script adapts its
standard MkDocs Markdown for GitHub Wiki without changing the source files:
``index.md`` becomes ``Home.md``, local page links lose their ``.md`` suffix,
and ``_Sidebar.md`` is generated from ``mkdocs.yml`` navigation.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CONFIG = ROOT / "mkdocs.yml"
GENERATED_NOTICE = "<!-- Generated from PESLite/docs; do not edit this Wiki page directly. -->"
MARKDOWN_LINK = re.compile(r"(?<!!)\]\((?P<target>[^)\s]+?\.md)(?P<anchor>#[^)\s]+)?\)")
WIKI_REMOTE = re.compile(r"(?:^|/)PESLite\.wiki(?:\.git)?$", re.IGNORECASE)


def _wiki_target(document: str) -> str:
    """Return the Wiki page target for one relative Markdown document."""
    path = PurePosixPath(document)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"documentation link leaves docs/: {document}")
    without_suffix = path.with_suffix("")
    if without_suffix.as_posix() == "index":
        return "Home"
    return without_suffix.as_posix()


def _rewrite_links(markdown: str) -> str:
    """Convert local MkDocs page links to GitHub Wiki page links."""
    def replace(match: re.Match[str]) -> str:
        target = match.group("target")
        if "://" in target or target.startswith("mailto:"):
            return match.group(0)
        return f"]({_wiki_target(target)}{match.group('anchor') or ''})"

    return MARKDOWN_LINK.sub(replace, markdown)


def _load_config() -> dict[str, Any]:
    with CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("nav"), list):
        raise ValueError("mkdocs.yml must contain a nav list")
    return config


def _nav_lines(items: list[Any], *, depth: int = 0) -> list[str]:
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"unsupported MkDocs navigation item: {item!r}")
        title, value = next(iter(item.items()))
        if isinstance(value, str):
            indent = "  " * depth
            lines.append(f"{indent}- [{title}]({_wiki_target(value)})")
        elif isinstance(value, list):
            if depth == 0:
                if lines and lines[-1] != "":
                    lines.append("")
                lines.extend((f"## {title}", ""))
                lines.extend(_nav_lines(value, depth=0))
            else:
                lines.append(f"{'  ' * depth}- {title}")
                lines.extend(_nav_lines(value, depth=depth + 1))
        else:
            raise ValueError(f"unsupported MkDocs navigation value: {value!r}")
    return lines


def _sidebar(config: dict[str, Any]) -> str:
    site_name = str(config.get("site_name", "Documentation"))
    lines = [GENERATED_NOTICE, "", f"## {site_name}", ""]
    lines.extend(_nav_lines(config["nav"]))
    while lines and lines[-1] == "":
        lines.pop()
    repo_url = config.get("repo_url")
    if repo_url:
        lines.extend(("", "---", "", f"[GitHub Repository]({repo_url})"))
    lines.extend(("", "[PyPI](https://pypi.org/project/peslite/)", ""))
    return "\n".join(lines)


def _validate_destination(destination: Path) -> None:
    destination = destination.resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), ROOT, ROOT.parent, DOCS}
    if destination in protected:
        raise ValueError(f"refusing to replace protected directory: {destination}")
    if not destination.exists() or not any(destination.iterdir()):
        return
    git_marker = destination / ".git"
    if not git_marker.exists():
        generated_home = destination / "Home.md"
        if generated_home.is_file() and generated_home.read_text(
            encoding="utf-8"
        ).startswith(GENERATED_NOTICE):
            return
        raise ValueError(
            f"destination is non-empty and is not a Git checkout: {destination}"
        )
    completed = subprocess.run(
        ["git", "-C", str(destination), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
    )
    remote = completed.stdout.strip().removesuffix("/")
    if not WIKI_REMOTE.search(remote):
        raise ValueError(f"destination is not the PESLite Wiki checkout: {remote}")


def _clear_destination(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in destination.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def render(destination: Path) -> list[Path]:
    """Replace *destination* with the generated Wiki tree and return its files."""
    _validate_destination(destination)
    config = _load_config()
    nav_documents: set[str] = set()

    def collect(items: list[Any]) -> None:
        for item in items:
            value = next(iter(item.values()))
            if isinstance(value, str):
                nav_documents.add(value)
            elif isinstance(value, list):
                collect(value)

    collect(config["nav"])
    source_documents = {path.relative_to(DOCS).as_posix() for path in DOCS.rglob("*.md")}
    missing = nav_documents - source_documents
    unlisted = source_documents - nav_documents
    if missing or unlisted:
        details = []
        if missing:
            details.append(f"missing from docs/: {sorted(missing)}")
        if unlisted:
            details.append(f"missing from mkdocs nav: {sorted(unlisted)}")
        raise ValueError("; ".join(details))

    _clear_destination(destination)
    written: list[Path] = []
    for source in sorted(DOCS.rglob("*")):
        if not source.is_file() or source.name == "requirements.txt":
            continue
        relative = source.relative_to(DOCS)
        if source.suffix.lower() == ".md":
            relative = Path("Home.md") if relative.as_posix() == "index.md" else relative
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            body = _rewrite_links(source.read_text(encoding="utf-8"))
            target.write_text(f"{GENERATED_NOTICE}\n\n{body}", encoding="utf-8")
        else:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        written.append(target)

    sidebar = destination / "_Sidebar.md"
    sidebar.write_text(_sidebar(config), encoding="utf-8")
    written.append(sidebar)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="empty directory or PESLite Wiki checkout")
    args = parser.parse_args()
    files = render(args.destination)
    print(f"generated {len(files)} Wiki files in {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
